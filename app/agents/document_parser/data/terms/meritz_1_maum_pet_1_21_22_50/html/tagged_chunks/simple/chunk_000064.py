from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>④ 회사는 제1항의 규정에 정한 지급기일내에 보험금을 지급하지 않았을 때(제2항의 "
 '규정<br>에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지의<br>기간에 대하여 <부표1> ‘보험금을 '
 '지급할 때의 적립이율 계산’에서 정한 이율로 계산<br>한 금액을 보험금에 더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
