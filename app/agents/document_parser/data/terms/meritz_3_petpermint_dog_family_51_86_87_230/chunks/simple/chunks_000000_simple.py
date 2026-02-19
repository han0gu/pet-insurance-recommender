from langchain_core.documents import Document

chunk = Document(
    page_content=('제1조(목적)\n'
 '이 보험계약(이하「계약」이라 합니다)은 보험계약자(이하 「계약자」라 합니다)와 보험회사(이하「회사」라 합니다) 사이에 피보험자의 상해에 '
 '대한 위험을 보장하기 위하여 체 결됩니다.\n'
 '제2조(용어의 정의)\n'
 '이 계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항 에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '\uf000 계약 관련 용어\n'
 '용어 | 정의\n'
 '계약자 | 회사와 계약을 체결하고 보험료를 납입할 의 무를 지는 사람을 말합니다.\n'
 '기본계약 | 계약자와 회사가 체결한 계약내용 중 보통약 관에 해당하는 부분을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 51},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000000',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
