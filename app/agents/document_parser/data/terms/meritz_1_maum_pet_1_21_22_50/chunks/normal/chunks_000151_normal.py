from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사는 제1항의 지급보험금이 결정된 후 7일(이하「지급기일」이라 합니다)이 지나도록 보험금을 지급하지 않았을 때에는 지급기일의 '
 '다음날부터 지급일까지의 기간에 대하여 <부표2> ‘보험금을 지급할 때의 적립이율 계산’에서 정한 이율로 계산한 금액을 보험금 에 더하여 '
 '지급합니다. 그러나 계약자 또는 피보험자의 책임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
