from langchain_core.documents import Document

chunk = Document(
    page_content=('기본계약 | 계약자와 회사가 체결한 계약내용 중 보통약 관에 해당하는 부분을 말합니다.\n'
 '보험 수익자 | 보험금 지급사유가 발생하는 때에 회사에 보 험금을 청구하여 받을 수 있는 사람을 말합니 다.\n'
 '보험증권 | 계약의 성립과 그 내용을 증명하기 위하여 회 사가 계약자에게 드리는 증서를 말합니다.\n'
 '진단계약 | 계약을 체결하기 위하여 피보험자가 건강진단 을 받아야 하는 계약을 말합니다.\n'
 '피보험자 | 보험사고의 대상이 되는 사람을 말합니다.\n'
 '\uf000 지급사유 관련 용어\n'
 '용어 | 정의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 51},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000001',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
