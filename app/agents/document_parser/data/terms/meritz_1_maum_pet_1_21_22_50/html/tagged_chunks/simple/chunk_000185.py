from langchain_core.documents import Document

chunk = Document(
    page_content=('지급사유가<br>발생한 후 3년간 보험금을 청구하지 않는 경우 보험금을 지급받지 못할 수 있습니다.<br>(이하 같습니다.)</p><h1 '
 "id='107' style='font-size:14px'>제37조(약관의 해석)</h1><br><p id='108' "
 "data-category='paragraph' style='font-size:14px'>① 회사는 신의성실의 원칙에 따라 공정하게 약관을 "
 "해석하여야 하며 계약자에 따라 다르<br>게 해석하지 않습니다.</p><br><h1 id='109'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000185',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
