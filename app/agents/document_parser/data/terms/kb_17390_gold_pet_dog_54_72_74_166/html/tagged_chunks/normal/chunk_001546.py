from langchain_core.documents import Document

chunk = Document(
    page_content=("추간판탈출증으로 인한 약간의 신경 장해</td><td>10</td></tr></tbody></table><h1 id='35' "
 "style='font-size:16px'>나.</h1><br><h1 id='36' "
 "style='font-size:16px'>장해판정기준</h1><br><p id='37' data-category='list' "
 "style='font-size:16px'>1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일한 부위로 한다.<br>제2천추 "
 '이하의 천골 및 미골은 체간골의 장해로 평가한다.<br>2)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001546',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
