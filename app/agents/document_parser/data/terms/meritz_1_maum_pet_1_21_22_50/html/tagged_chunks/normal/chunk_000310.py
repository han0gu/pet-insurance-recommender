from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3종 단체</h1><br><p id='56' data-category='paragraph' "
 "style='font-size:14px'>그 밖에 단체의 구성원을 확정시킬 수 있고 계약의 일괄적인 관리가 가능한 단체로<br>서 5인 "
 "이상의 구성원이 있는 단체</p><br><p id='57' data-category='paragraph' "
 "style='font-size:14px'>② 제1항의 대상단체에 소속된 자로서 동일한 보험계약을 체결한 5인 이상의 "
 '피보험자로<br>피보험단체를 구성하여야 하며, 단체 구성원의 일부만을 대상으로'),
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
 'indexing': {'chunk_id': 'chunk_000310',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
