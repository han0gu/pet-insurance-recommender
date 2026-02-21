from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우에는 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.\n'
 '【보험안내자료】 계약의 청약을 권유하기 위해 만든 서류 등을 말합니다.제36조(회사의 손해배상책임)- ① 회사는 계약과 관련하여 임직원, '
 '보험 설계사 및 대리점의 책임있는 사유로 인하여 계약자 및 피보\n'
 '- 험자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상의 책임을 집니다.\n'
 '- ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었음에도 불구하고 소를 제'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
