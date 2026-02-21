from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라\n'
 '회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.<용어풀이># [가지급보험금]보험금 지급이 늦어지는 경우 보험수익자 '
 '청구에 따라 확정된 보험금을 먼저 지급하는 제도\n'
 '[장해지급률]# 질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.- ④ 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지급하지 않았을 때(제2항의 규'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
