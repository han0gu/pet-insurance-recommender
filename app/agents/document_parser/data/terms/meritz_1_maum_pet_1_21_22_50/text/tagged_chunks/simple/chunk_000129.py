from langchain_core.documents import Document

chunk = Document(
    page_content=('일 이내에 이를 지급하여 드립니다. 또한, 지급할 보험금이 결정되기 전이라도 피보험\n'
 '자의 청구가 있을 때에는 회사가 추정한 보험금의 50% 상당액을 가지급보험금으로 지\n'
 '급합니다.\n'
 '② 회사는 제1항의 지급보험금이 결정된 후 7일(이하「지급기일」이라 합니다)이 지나도록\n'
 '보험금을 지급하지 않았을 때에는 지급기일의 다음날부터 지급일까지의 기간에 대하여\n'
 '<부표2> ‘보험금을 지급할 때의 적립이율 계산’에서 정한 이율로 계산한 금액을 보험금\n'
 '에 더하여 지급합니다. 그러나 계약자 또는 피보험자의 책임있는 사유로 지급이 지연된'),
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
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
