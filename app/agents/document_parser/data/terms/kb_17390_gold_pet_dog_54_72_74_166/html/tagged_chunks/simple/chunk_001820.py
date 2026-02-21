from langchain_core.documents import Document

chunk = Document(
    page_content=('선천성 피부질환</td></tr><tr><td>원인 불명의 귀 '
 '소양감</td></tr><tr><td>탈모</td></tr><tr><td>원인 불명의 피부 '
 "소양감</td></tr></tbody></table><p id='10' data-category='paragraph' "
 "style='font-size:14px'>166 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='11' "
 "data-category='paragraph' style='font-size:18px'>- 166 -</p>"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_001820',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
