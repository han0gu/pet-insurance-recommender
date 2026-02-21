from langchain_core.documents import Document

chunk = Document(
    page_content=("어 풀 이</td><td>질</td></tr></tbody></table><br><p id='70' "
 "data-category='paragraph' style='font-size:16px'>강제집행</p><br><p id='71' "
 "data-category='paragraph' style='font-size:16px'>∙</p><br><p id='72' "
 "data-category='paragraph' style='font-size:16px'>강제집행이란 사법상 또는 행정법상의 의무를 "
 '이행하지 않는 사람에 대하여 국<br>가가 강제'),
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
 'indexing': {'chunk_id': 'chunk_000899',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
