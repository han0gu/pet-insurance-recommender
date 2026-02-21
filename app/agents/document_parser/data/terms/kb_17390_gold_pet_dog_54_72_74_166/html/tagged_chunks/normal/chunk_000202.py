from langchain_core.documents import Document

chunk = Document(
    page_content=('산정하는 기준이 되는 보험가입금액을 계 약시 선택한 금액보다 적은 금액으로 줄이는 것 (이에 따라 보험료, 보험금 '
 '및</td></tr><tr><td>적립액(해약환급금)도</td><td>줄어듭니다.)</td></tr><tr><td '
 'colspan="2">용 어 풀 이 계약자적립액 장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 '
 "중</td></tr></tbody></table><br><p id='15' data-category='paragraph' "
 "style='font-size:16px'>일정액을 회사가 적립해"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000202',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
