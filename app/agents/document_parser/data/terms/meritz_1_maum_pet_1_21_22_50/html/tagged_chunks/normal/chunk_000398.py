from langchain_core.documents import Document

chunk = Document(
    page_content=('61일이후부터 90일이내 기간</td><td>보험계약대출이율+ 가산이율(6.0%)</td></tr><tr><td>지급기일의 91일이후 '
 "기간</td><td>보험계약대출이율+ 가산이율(8.0%)</td></tr></tbody></table><br><p id='89' "
 "data-category='paragraph' style='font-size:14px'>주) 보험계약대출이율은 보험개발원이 공시하는 "
 "보험계약대출이율을 적용합니다.</p><footer id='90' style='font-size:14px'>- 49"),
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
 'indexing': {'chunk_id': 'chunk_000398',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
