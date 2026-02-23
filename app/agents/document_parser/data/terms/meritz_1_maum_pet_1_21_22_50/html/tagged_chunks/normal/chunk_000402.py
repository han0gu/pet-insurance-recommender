from langchain_core.documents import Document

chunk = Document(
    page_content=("95%</td></tr></tbody></table><br><p id='93' data-category='list' "
 "style='font-size:14px'>주1) 연요율 : 기본요율 및 특별약관요율에 소정의 할증 및 할인요율을 가감한 요율<br>주2) "
 '보험기간이 1년미만인 단기계약에 대하여는 위 단기요율을 적용한다.<br>주3) 보험기간을 연장하는 경우에는 원기간에 통산 아니하고 그 '
 "연장기간에 대한 단기요율<br>을 적용한다.</p><footer id='94' style='font-size:14px'>- 50 "
 '-</footer>'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000402',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
