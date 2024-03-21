import api_recon_intencao

res = api_recon_intencao.call_model_qa('Me envie os processos abertos ha 3 meses', 'Quanto tempo?')
print(res)

res2 = api_recon_intencao.call_model_zsc('Me envie os processos abertos ha 3 meses', ['maria da penha', 'processos abertos'])
print(res2)