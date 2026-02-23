from langchain_core.documents import Document

chunk = Document(
    page_content=('나타낸 것을 말합니다.# ④ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라# 회사가 추정하는 보험금의 '
 '50% 상당액을 가지급보험금으로 지급합니다.<용어풀이># [가지급보험금]# 보험금 지급이 늦어지는 경우 보험수익자 청구에 따라 확정된 '
 '보험금을 먼저 지급하는 제도- ⑤ 회사는 제1항의 규정에 정한 지급기일내에 보험금을 지급하지 않았을 때(제2항의 규\n'
 '- 정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
