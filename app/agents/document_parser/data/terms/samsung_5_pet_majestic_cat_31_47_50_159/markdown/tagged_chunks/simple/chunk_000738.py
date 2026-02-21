from langchain_core.documents import Document

chunk = Document(
    page_content=('별할 수 있을 정도의 시력상태\n'
 '주2) 안전수지 : 시표의 가장 큰 글씨를 읽을 수 있는 정도의 시력은 아니나 눈 앞\n'
 '30cm 이내에서 손가락의 개수를 식별할 수 있을 정도의 시력상태- 5) 안구(눈동자) 운동장해의 판정은 질병의 진단 또는 외상 후 1년 '
 '이상이 지난 뒤\n'
 '- 그 장해 정도를 평가한다.\n'
 '- 6) "안구(눈동자)의 뚜렷한 운동장해" 라 함은 아래의 두 경우 중 하나에 해당하\n'
 '- 는 경우를 말한다.\n'
 '- 가) 한 눈의 안구(눈동자)의 주시야(머리를 움직이지 않고 눈만을 움직여서 볼'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000738',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
