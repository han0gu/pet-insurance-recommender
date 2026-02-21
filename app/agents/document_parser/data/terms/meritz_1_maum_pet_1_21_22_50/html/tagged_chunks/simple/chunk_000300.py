from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제3조(보험료의 영수)</p><br><p id='34' "
 "data-category='paragraph' style='font-size:14px'>자동납입일자는 이 보험계약 청약서에 기재된 보험료 "
 "납입 해당일에도 불구하고 매월 회사<br>가 정하는 날 중 보험계약자가 희망하는 일자로 합니다.</p><p id='35' "
 "data-category='paragraph' style='font-size:14px'>제4조(계약 후 알릴 의무)</p><br><p "
 "id='36'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000300',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
