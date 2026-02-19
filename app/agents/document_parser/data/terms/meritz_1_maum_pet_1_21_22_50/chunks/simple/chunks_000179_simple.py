from langchain_core.documents import Document

chunk = Document(
    page_content=('중요한 사항에 해당되는 사유를「반대증거가 있는 경우 이의를 제기할 수 있습니다」 라는 문구와 함께 계약자에게 서면 또는 전자문서 등으로 '
 '알려 드립니다. 또한 이 경우 계약 해지로 인하여 회사가 환급하여야 할 보험료가 있을 때에는 보통약관 제33조(보 험료의 환급)에 따른 '
 '보험료를 계약자에게 지급합니다. 회사가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 '
 '전자서명으로 동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 29},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000179',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
