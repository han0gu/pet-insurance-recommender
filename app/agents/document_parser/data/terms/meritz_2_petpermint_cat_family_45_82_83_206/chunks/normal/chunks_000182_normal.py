from langchain_core.documents import Document

chunk = Document(
    page_content=('Ⅰ. 반려동물 비용손해 관련 특별약관\n'
 '반려동물 비용손해 관련 특별약관 일반조항\n'
 '제1조(목적)\n'
 '이 특별약관은 계약자와 회사 사이에 피보험자 소유의 보험 증권에 기재된 반려동물의 질병 또는 상해로 인한 손해를 보장하기 위하여 '
 '체결됩니다.\n'
 '제2조(용어의 정의)\n'
 '이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '\uf000 계약관련 용어\n'
 '용어 | 정의\n'
 '계약자 | 회사와 계약을 체결하고 보험료를 납입할 의무 를 지는 사람을 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
