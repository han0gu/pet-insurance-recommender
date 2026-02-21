from langchain_core.documents import Document

chunk = Document(
    page_content=('함을 원칙으로 한다. 단, 질병발생 또는 상해를\n'
 '입은 후 의식상실이 1개월 이상 지속된 경우에\n'
 '는 질병발생 또는 상해를 입은 후 12개월이 지난\n'
 '후에 판정할 수 있다.- 나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정\n'
 '- 신건강의학과의 전문적 치료를 받은 후 치료에도\n'
 '- 불구하고 장해가 고착되었을 때 판정하여야 하며,\n'
 '- 그렇지 않은 경우에는 그로써 고정되거나 중하게\n'
 '- 된 장해에 대해서는 인정하지 않는다.\n'
 '- 다) “정신행동에 극심한 장해를 남긴 때”라 함은 장\n'
 '- 해판정 직전 1년 이상 지속적인 정신건강의학과의'),
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
 'indexing': {'chunk_id': 'chunk_000685',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
