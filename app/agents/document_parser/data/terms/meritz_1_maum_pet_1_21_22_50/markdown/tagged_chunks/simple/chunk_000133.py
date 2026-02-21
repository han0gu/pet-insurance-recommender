from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서류를 접수받은 후 지체없이 지급할 보험금을 결정하고 지급할 보험금이 결정되면 7\n'
 '- 일 이내에 이를 지급하여 드립니다. 또한, 지급할 보험금이 결정되기 전이라도 피보험\n'
 '- 자의 청구가 있을 때에는 회사가 추정한 보험금의 50% 상당액을 가지급보험금으로 지\n'
 '- 급합니다.\n'
 '- ② 회사는 제1항의 지급보험금이 결정된 후 7일(이하「지급기일」이라 합니다)이 지나도록\n'
 '- 보험금을 지급하지 않았을 때에는 지급기일의 다음날부터 지급일까지의 기간에 대하여'),
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
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
