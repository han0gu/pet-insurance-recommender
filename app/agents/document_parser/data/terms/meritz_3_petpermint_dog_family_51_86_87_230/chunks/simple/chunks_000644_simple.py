from langchain_core.documents import Document

chunk = Document(
    page_content=('Ⅲ. 기타 특별약관\n'
 '1. 자동갱신 특별약관\n'
 '제1조(특별약관의 적용)\n'
 '\uf000 이 특별약관은 무배당 펫퍼민트 Puppy&Family보험 다이 렉트2601 특별약관 중 자동갱신 적용대상 특별약관(이하 '
 '「자동갱신 적용대상 계약」이라 합니다)의 자동갱신에 대 하여 회사와 계약자간에 합의가 되었을 경우에 적용합니다. \uf000 제1항의 '
 '자동갱신 적용대상 특별약관(이하「자동갱신 적 용대상 특별약관」이라 합니다)이라 함은 아래의 특별약관 을 말합니다.\n'
 '【자동갱신 적용대상 특별약관】\n'
 '･ 갱신형 펫퍼민트 반려견 배상책임보장 특별약관'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 189},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000644',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
