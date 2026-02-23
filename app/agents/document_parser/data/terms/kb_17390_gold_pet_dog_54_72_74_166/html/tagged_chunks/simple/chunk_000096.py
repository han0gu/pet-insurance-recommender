from langchain_core.documents import Document

chunk = Document(
    page_content=("일정한 기간동안(예: 6개월 이상) 계속하여 종사</p><br><p id='120' data-category='paragraph' "
 "style='font-size:16px'>하는 일을 말합니다.</p><br><p id='121' "
 "data-category='paragraph' style='font-size:16px'>2) 1)에 해당하지 않는 경우에는 개인의 사회적 "
 "신분에 따르는 위치나 자리</p><br><p id='122' data-category='paragraph' "
 "style='font-size:16px'>를"),
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
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
