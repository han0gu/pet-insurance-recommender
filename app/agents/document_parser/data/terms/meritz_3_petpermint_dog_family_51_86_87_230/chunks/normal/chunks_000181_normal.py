from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무 를 지는 사람을 말합니다.\n'
 '보험 수익자 | 보험금 지급사유가 발생하는 때에 회사에 보험 금을 청구하여 받을 수 있는 사람을 말합니다.\n'
 '보험증권 | 계약의 성립과 그 내용을 증명하기 위하여 회 사가 계약자에게 드리는 증서를 말합니다.\n'
 '진단계약 | 계약을 체결하기 위하여 반려동물이 건강진단 을 받아야 하는 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
