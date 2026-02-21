from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 중 일정액을 기준으로 보험료 및 해약환급금 산출방법서에서 정한 방법에 따라 계산한 금액을 '
 '말합니다.</td></tr><tr><td>해약 환급금</td><td>계약이 해지되는 때에 회사가 계약자에게 돌 려주는 금액을 '
 "말합니다.</td></tr></tbody></table><br><h1 id='18' style='font-size:20px'>【연단위 "
 "복리 】</h1><br><p id='19' data-category='paragraph' style='font-size:20px'>회사가 "
 '지급할 금전에 이자를 줄 때, 1년마다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
