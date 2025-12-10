import time
from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest local accuracy.")
        print(max(self.rs_test_acc) if len(self.rs_test_acc) > 0 else 0.0)

        # Also report the best global accuracy if available
        try:
            best_global = max(self.rs_global_acc) if hasattr(self, 'rs_global_acc') and len(self.rs_global_acc) > 0 else None
        except Exception:
            best_global = None
        print("\nBest global accuracy.")
        print(best_global if best_global is not None else 0.0)

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else (self.Budget[0] if len(self.Budget) > 0 else 0.0))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMOON)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
