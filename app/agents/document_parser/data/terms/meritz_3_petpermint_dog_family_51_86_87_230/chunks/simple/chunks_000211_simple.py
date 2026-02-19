from langchain_core.documents import Document

chunk = Document(
    page_content=('① 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때 ② 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 '
 '다른 보험자와 체결하고자 할 때 또는 이와 같 은 계약이 있음을 알았을 때 ③ 반려동물을 양도할 때 ④ 위 이외에 위험이 뚜렷이 '
 '변경되거나 변경되었음을 알 았을 때\n'
 '\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경 우에는 제13조(계약내용의 변경 등)에 따라 계약내용을 변 경할 수 '
 '있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 95},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
