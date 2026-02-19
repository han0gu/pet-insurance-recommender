from langchain_core.documents import Document

chunk = Document(
    page_content=('제 40조 (소멸시효)\n'
 '보험금청구권, 보험료 반환청구권, 해약환급금 청구권, 계약자적립액 및 미경과보험료 반 환청구권은 3년간 행사하지 않으면 소멸시효가 '
 '완성됩니다.\n'
 '<용어풀이>\n'
 '[소멸시효]\n'
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합니다. 보험금 지급사유가 2021년 4월 1일에 발생하였음에도 2024년 4월 '
 '1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 보험금 등을 지급받지 못할 수 있습니다.\n'
 '제 41조 (약관의 해석)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000300',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
