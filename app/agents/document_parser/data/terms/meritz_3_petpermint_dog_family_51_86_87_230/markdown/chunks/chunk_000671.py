from langchain_core.documents import Document

chunk = Document(
    page_content=('- 들면 캐스트로 환부를 고정시켰기 때문에 치유 후의 관\n'
 '222절에 기능장해가 발생한 경우)는 장해로 평가하지 않\n'
 '는다.- 3) “발가락을 잃었을 때”라 함은 첫째 발가락에서는\n'
 '- 지관절부터 심장에 가까운 쪽을, 나머지 네 발가락에\n'
 '- 서는 제1지관절(근위지관절)부터(제1지관절 포함) 심\n'
 '- 장에서 가까운 쪽을 잃었을 때를 말한다.\n'
 '- 4) 리스프랑 관절 이상에서 잃은 때라 함은 족근-중족골\n'
 '- 간 관절 이상에서 절단된 경우를 말한다.\n'
 '- 5) “발가락뼈 일부를 잃었을 때”라 함은 첫째 발가락'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
