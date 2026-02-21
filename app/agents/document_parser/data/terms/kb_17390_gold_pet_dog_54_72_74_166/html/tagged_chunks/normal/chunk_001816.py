from langchain_core.documents import Document

chunk = Document(
    page_content=("선천성 안과질환 규정</td></tr><tr><td>기타 안과 질환</td></tr></tbody></table><p id='7' "
 "data-category='paragraph' style='font-size:14px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 165</p><br><p id='8' data-category='paragraph' "
 "style='font-size:18px'>- 165 -</p><table id='9' "
 "style='font-size:14px'><thead><tr><td></td><td>코드"),
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
 'indexing': {'chunk_id': 'chunk_001816',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
