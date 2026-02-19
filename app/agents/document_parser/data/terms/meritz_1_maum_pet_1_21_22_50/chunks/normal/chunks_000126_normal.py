from langchain_core.documents import Document

chunk = Document(
    page_content=('제36조(소멸시효)\n'
 '보험금청구권, 보험료 또는 환급금 반환청구권은 3년간 행사하지 않으면 소멸시효(소멸시 효는 해당 청구권을 행사할 수 있는 때로부터 '
 '진행합니다.)가 완성됩니다.\n'
 '【소멸시효】\n'
 '주어진 권리를 행사하지 않을 때 그 권리가 없어지게 되는 기간으로 보험금 지급사유가 발생한 후 3년간 보험금을 청구하지 않는 경우 '
 '보험금을 지급받지 못할 수 있습니다. (이하 같습니다.)\n'
 '제37조(약관의 해석)\n'
 '① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다르 게 해석하지 않습니다.\n'
 '【신의성실의 원칙】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000126',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
