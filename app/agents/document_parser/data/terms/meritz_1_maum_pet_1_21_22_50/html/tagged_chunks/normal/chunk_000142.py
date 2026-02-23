from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단<br>계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는 경우에는<br>보장을 '
 "해드립니다.</p><br><p id='47' data-category='paragraph' style='font-size:14px'>④ "
 '계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종<br>료하는 때부터 적용합니다.</p><h1 '
 "id='48' style='font-size:14px'>제26조(제2회 이후 보험료의 납입)</h1><br><p id='49'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
