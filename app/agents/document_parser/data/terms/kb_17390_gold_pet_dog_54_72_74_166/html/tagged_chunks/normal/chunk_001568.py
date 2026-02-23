from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td></td></tr></thead><tbody><tr><td>장해의 분류 1) 어깨뼈(견갑골)나 '
 '골반뼈(장골, 제2천추 이하의 천골, 미골, 좌골 포함)에 뚜렷한 기형을 남긴 때</td><td>지급률 '
 '15</td></tr><tr><td>2) 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)에 뚜렷한 '
 "기형을</td><td>10</td></tr></tbody></table><br><h1 id='69' "
 "style='font-size:14px'>남긴 때</h1><p id='70'"),
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
 'indexing': {'chunk_id': 'chunk_001568',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
