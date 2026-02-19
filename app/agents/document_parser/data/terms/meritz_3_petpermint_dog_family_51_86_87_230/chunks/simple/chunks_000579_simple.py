from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무 를 지는 사람을 말합니다.\n'
 '보험증권 | 계약의 성립과 그 내용을 증명하기 위하여 회 사가 계약자에게 드리는 증서를 말합니다.\n'
 '피보험자 | 보험사고로 인하여 타인에 대한 법률상 손해배 상책임을 부담하는 손해를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무를 '
 '집행하 는 그 밖의 기관)을 말합니다.\n'
 '\uf000 알릴의무 관련 용어\n'
 '용어 | 정의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 174},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000579',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
