from langchain_core.documents import Document

chunk = Document(
    page_content=('된 경우로서 아래의 의료행위를 말합니다.\n'
 '제\n'
 '1. "자기공명영상(MRI)"이란 수의사의 관리 하에 자기공명영상(MRI)을 사용하는\n'
 '도- 촬영 의료행위를 말합니다.\n'
 '- 성\n'
 '- 2. "컴퓨터단층촬영(CT)"이란 수의사의 관리 하에 자기공명영상(MRI)을 사용하 특\n'
 '- 는 촬영 의료행위를 말합니다. 약\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 113- 113 -병|  |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000591',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
