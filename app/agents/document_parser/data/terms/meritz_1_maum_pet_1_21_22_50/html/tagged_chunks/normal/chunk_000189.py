from langchain_core.documents import Document

chunk = Document(
    page_content=('청약서 및 보험증권의 제공 사실에 관하여 계약자와 회사<br>간에 다툼이 있는 경우에는 회사가 이를 증명하여야 합니다.<br>③ '
 '보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료(계약의 청약을 권유하<br>기 위해 만든 자료 등을 말합니다)의 내용이 '
 "약관의 내용과 다른 경우에는 계약자에게<br>유리한 내용으로 계약이 성립된 것으로 봅니다.</p><h1 id='114' "
 "style='font-size:14px'>제39조(회사의 손해배상책임)</h1><br><p id='115' "
 "data-category='list'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
