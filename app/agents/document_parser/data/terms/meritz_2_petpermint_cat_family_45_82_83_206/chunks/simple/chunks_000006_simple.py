from langchain_core.documents import Document

chunk = Document(
    page_content=('【한국표준질병사인분류 부호 체계】\n'
 '질병의 원인과 증상 두 가지 모두에 관한 정보를 포함 하는 진단을 위해 아래 두 가지 분류부호가 사용됩니 다. 또한 원인과 질환에 따라 '
 '동시에 사용될 수 있습니 다.\n'
 '- 검표(+) : 원인이 되는 질환에 대한 질병분류코드 - 별표(*) : 원인(검표)으로 인한 발현증세에 대한 질 병분류코드\n'
 '\uf000 지급금과 이자율 관련 용어\n'
 '용어 | 정의\n'
 '연단위 복리 | 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 이자 '
 '계산방법을 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
