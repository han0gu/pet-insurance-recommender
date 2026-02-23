from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인이 되\n'
 '는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. 단, 제거가 불가능\n'
 '한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.\n'
 '2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를 들면 캐스트로 환부를 고\n'
 '정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는 장해로 평가하\n'
 '지 않는다.\n'
 '3) "다리" 라 함은 엉덩이관절(고관절)부터 발목관절(족관절)까지를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000799',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
