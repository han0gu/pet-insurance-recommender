from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을 '
 '조건 으로 전자문서를 송신하여야 합니다. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 '
 '전자문서가 수신되지 않은 것을 확인 한 경우에는 서면(등기우편 등)으로 다시 알려드립니다. ⑤ 제1항 제2호에 의한 계약의 해지가 보험금 '
 '지급사유 발생 후에 이루어진 경우에는 제 16조(계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
