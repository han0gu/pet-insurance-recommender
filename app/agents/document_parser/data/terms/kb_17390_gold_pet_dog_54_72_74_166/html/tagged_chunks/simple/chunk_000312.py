from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험금 지급사유 가 2022년 9월 1일에 발생하였음에도 2025년 9월 1일까지 보험금을 청구하지 '
 "않는</td></tr></tbody></table><br><h1 id='160' style='font-size:14px'>경우 소멸시효가 "
 "완성되어 보험금 등을 지급받지 못할 수 있습니다.</h1><p id='161' data-category='list' "
 "style='font-size:14px'>제46조(약관의 해석)<br>\uf000 회사는 신의성실의 원칙에 따라 공정하게 약관을 "
 '해석하여야 하며 계약자에 따라<br>다르게 해석하지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000312',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
