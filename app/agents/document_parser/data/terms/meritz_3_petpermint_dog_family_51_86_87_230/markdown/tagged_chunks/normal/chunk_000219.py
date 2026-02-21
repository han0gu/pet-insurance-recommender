from langchain_core.documents import Document

chunk = Document(
    page_content=('분 절차에 따라 회사는 채권자에게 해약환급금을 지급하\n'
 '게 됩니다.# 제20조(계약자의 임의해지)계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할\n'
 '수 있으며, 이 경우 회사는 보통약관 제35조(해약환급금)\n'
 '제1항에 의한 해약환급금을 계약자에게 지급합니다. 다만,\n'
 '타인을 위한 계약의 경우에는 계약자는 그 타인의 동의를\n'
 '얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 수\n'
 '있습니다.# 제21조(중대사유로 인한 해지)\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000219',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
