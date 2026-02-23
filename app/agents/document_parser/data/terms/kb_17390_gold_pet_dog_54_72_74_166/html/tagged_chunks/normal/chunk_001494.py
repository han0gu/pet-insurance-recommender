from langchain_core.documents import Document

chunk = Document(
    page_content=('특별 약</td></tr><tr><td>5) 한 귀의 청력에 약간의 장해를 남긴 때</td><td>5 '
 '관</td></tr><tr><td>6) 한 귀의 귓바퀴의 대부분이 결손된 때</td><td>10</td></tr><tr><td>7) '
 "평형기능에 장해를 남긴 때</td><td>10</td></tr></tbody></table><p id='183' "
 "data-category='paragraph' style='font-size:16px'>나"),
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
 'indexing': {'chunk_id': 'chunk_001494',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
