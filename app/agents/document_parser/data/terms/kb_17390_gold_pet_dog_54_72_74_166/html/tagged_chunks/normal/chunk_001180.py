from langchain_core.documents import Document

chunk = Document(
    page_content=(". 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.</td></tr></tbody></table><br><h1 id='197' "
 "style='font-size:14px'>제12조(손해방지의무)</h1><br><p id='198' "
 "data-category='paragraph' style='font-size:14px'>계약자 또는 피보험자는 아래의 사항을 이행하여야 "
 "합</p><br><p id='199' data-category='paragraph' style='font-size:14px'>\uf000 "
 '보험사고가 생긴'),
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
 'indexing': {'chunk_id': 'chunk_001180',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
