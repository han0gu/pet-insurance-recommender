from langchain_core.documents import Document

chunk = Document(
    page_content=("보상했을 금액을 제1항의 의무보험에서 보상하는 금액으로 봅<br>니다.</p><h1 id='58' "
 "style='font-size:14px'>제10조(보험금의 분담)</h1><br><p id='59' "
 "data-category='paragraph' style='font-size:14px'>① 회사는 이 특별약관에서 보장하는 위험과 같은 "
 '위험을 보장하는 다른 계약(공제계약을<br>포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 '
 '산출한<br>보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
