from langchain_core.documents import Document

chunk = Document(
    page_content=('되는 항목 수가코드 주: 길이 10cm이상 창상봉합술을 시행할경우 소정</td></tr></tbody></table><br><p '
 "id='71' data-category='paragraph' style='font-size:14px'>점수에 63.81점을 가산하며, "
 "창상봉합 길이가 SA040</p><br><h1 id='72' style='font-size:14px'>5cm 증가될때마다 63.81점을 "
 "추가 가산한다.</h1><p id='73' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001746',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
