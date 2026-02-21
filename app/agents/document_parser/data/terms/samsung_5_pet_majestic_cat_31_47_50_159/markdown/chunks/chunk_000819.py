from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.\n'
 '- 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를 들면 캐스트로 환부를 고\n'
 '- 정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는 장해로 평가하\n'
 '- 지 않는다.\n'
 '- 3) "발가락을 잃었을 때" 라 함은 첫째 발가락에서는 지관절부터 심장에 가까운\n'
 '- 쪽을, 나머지 네 발가락에서는 제1지관절(근위지관절)부터(제1지관절 포함) 심\n'
 '- 장에서 가까운 쪽을 잃었을 때를 말한다.\n'
 '- 4) 리스프랑 관절 이상에서 잃은 때라 함은 족근-중족골간 관절 이상에서 절단된'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
