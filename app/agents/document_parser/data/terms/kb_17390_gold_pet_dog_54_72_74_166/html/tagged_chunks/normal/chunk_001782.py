from langchain_core.documents import Document

chunk = Document(
    page_content=('상세불명 질환 급성인지 만성인지 명시되지 않은 기관지염</td><td>J39.9 J40</td></tr><tr><td>단순성 및 '
 '점액화농성 만성기관지염</td><td>J41</td></tr><tr><td>상세불명의 만성 '
 '기관지염</td><td>J42</td></tr><tr><td>천식, 천식지속 상태</td><td>J45, '
 'J46</td></tr><tr><td>폐렴</td><td>J12~J18</td></tr><tr><td>[B01.2+:'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001782',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
