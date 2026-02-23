from langchain_core.documents import Document

chunk = Document(
    page_content=('. 대상상병 분류표의 분류번호와 다르나 한국표준질병․사인분류의</td><td>기준에 '
 "따라</td></tr></tbody></table><br><p id='37' data-category='paragraph' "
 "style='font-size:16px'>분류번호를 동시에 부여가 가능한 경우 대상상병 분류에 포함합니다.<br>2"),
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
 'indexing': {'chunk_id': 'chunk_001708',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
