from langchain_core.documents import Document

chunk = Document(
    page_content=("id='71' style='font-size:14px'>제7조(적용상의 특칙)</h1><br><p id='72' "
 "data-category='paragraph' style='font-size:14px'>계약자가 아닌 단체의 소속원이 보험료 전부 또는 "
 "일부를 부담하는 경우에는 그 소속원이<br>계약자로서의 권리를 행사할 수 있습니다.</p><h1 id='73' "
 "style='font-size:14px'>제8조(준용규정)</h1><br><p id='74' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000323',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
