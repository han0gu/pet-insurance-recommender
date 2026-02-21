from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을 해<br>지할 수 있습니다.<br>④ 제1항 및 제3항에 '
 '따라 계약이 해지된 경우 회사는 제33조(보험료의 환급) 제1항 제1<br>호에 따른 보험료를 계약자에게 지급합니다.<br>⑤ 계약자는 '
 '제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에 따<br>라 법률상의 권리를 행사할 수 있습니다.</p><h1 '
 "id='83' style='font-size:14px'>제31조(중대사유로 인한 해지)</h1><br><p id='84'"),
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
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
