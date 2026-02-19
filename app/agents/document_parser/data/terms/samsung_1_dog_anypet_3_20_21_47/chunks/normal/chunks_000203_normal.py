from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 피보험자의 증감이 있을 경우 아래 [양식1]에 정한 양식으로 회사에 서면(팩시밀리를 포 함합니다)통지하여야 합니다. ② '
 '회사의 보장은 제1항의 통지가 회사에 접수되는 시점으로 하며 우편통지 시 그 통지가 지연된 경 우에는 우체국 소인이 찍힌 날로부터 3일이 '
 '지나면 회사에 접수된 것으로 봅니다. ③ 제1항에도 불구하고 계약자가 자기의 재화, 용역 및 서비스를 판매한 날짜 및 시간이 입력된 '
 'M/T 등 전산자료를 회사에 제공할 수 있을 경우에는 다음 어느 하나의 기간단위로 피보험자 증감내역 을 통보합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 41},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
