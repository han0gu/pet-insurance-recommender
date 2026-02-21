from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험단체를 구성하여야 하며, 단체 구성원의 일부만을 대상으로 가입하는 경우에는\n'
 '다음의 조건을 모두 충족하여야 합니다.- 1. 단체의 내규에 의한 복지제도로서 노사합의에 의하며, 보험료의 일부를 단체 또는 단\n'
 '- 체의 대표자가 부담하여야 합니다.\n'
 '- 2. 제1항 제2호 및 제3호에 해당하는 단체는 내규에 의해 단체의 대표자와 보험회사가\n'
 '- 협정에 의해 체결하여야 합니다.\n'
 '# 제2조(상법 제735조3의 적용)- ① 제1조(계약의 적용 범위)에 해당하는 단체가 피보험자를 확정할 수 있고 계약의 일괄적 관'),
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
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
